import img from "../../assets/new/menu.svg";
import Button from "../Button";
import { useEffect, useRef, useState } from "react";
import Menu from "./Menu";

const MenuButt = () => {
    const [isActive, setIsActive] = useState<boolean>(false);
    const menuRef = useRef<HTMLDivElement | null>(null);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (isActive && menuRef.current && !menuRef.current.contains(event.target as Node)) {
                setIsActive(false);
            }
        };

        if (isActive) {
            document.addEventListener("click", handleClickOutside);
        }

        return () => {
            document.removeEventListener("click", handleClickOutside);
        };
    }, [isActive]);

    const dynamicStyle = {
        transform: isActive ? 'rotateZ(180deg)' : 'rotateZ(0deg)',
        transition: 'transform 0.3s ease',
        height: '24px',
        zIndex: 110,
        
    };

    const MenuDynamicStyle = {
        transform: isActive ? 'translateX(0)' : 'translateX(-100%)',
    };

    const tap = (event?: MouseEvent)=>{
        setIsActive(!isActive);
        if(event){
            event.stopPropagation();
        }
    }    

    return(
        <>
            <div style={dynamicStyle}>
                <Button img={img} tap={tap}  alt={"menu"}/>
                
            </div>
            
            <Menu ref={menuRef} tap={tap} style={MenuDynamicStyle}/>
        </>
        
    );
};

export default MenuButt;